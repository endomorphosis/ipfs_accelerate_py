"""Bounded, content-addressed scans for prompt-driven repository workflows.

The scanner is deliberately a policy boundary rather than a recursive source
loader.  It resolves one requested directory beneath an explicit repository
allowlist, selects policy-admitted files, and delegates exact HEAD/index/
worktree and Python AST identities to :mod:`program_behavior`.  Durable scan
records contain only bounded summaries and content-addressed manifests; source
bodies are transient and are never placed in a receipt or evidence artifact.

Optional analysis is adapter-driven and lazy.  Importing this module does not
load a provider, retrieval backend, cache, model, or supervisor process.
Optional results remain ``scan_advisory`` and cannot acquire completion
authority.
"""

from __future__ import annotations

import fnmatch
import hashlib
import inspect
import json
import os
import re
import stat
import subprocess
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final, Protocol

from .prompt_workflow import (
    DirectoryScanReceipt,
    EvidenceAuthority,
    PromptEvidenceRecord,
    PromptWorkflowBudget,
    PromptWorkflowRequest,
    RecordStatus,
    canonical_prompt_workflow_bytes,
    prompt_workflow_cid,
)

if TYPE_CHECKING:
    from ..runtime.artifact_store import BoundedArtifactStore
    from ..core.program_behavior import ProgramBehavior


PROMPT_DIRECTORY_SCANNER_VERSION: Final[str] = "1.0.0"
SCANNER_VERSION: Final[str] = PROMPT_DIRECTORY_SCANNER_VERSION
REPOSITORY_ALLOWLIST_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/repository-allowlist@1"
)
REPOSITORY_ROOT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/resolved-repository-root@1"
)
SCAN_CONFIGURATION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/prompt-directory-scan-configuration@1"
)
SCAN_DECISION_MANIFEST_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/prompt-directory-scan-decisions@1"
)
SCAN_EVIDENCE_ARTIFACT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/prompt-directory-scan-evidence@1"
)
MAX_EVIDENCE_ARTIFACT_BYTES: Final[int] = 16 * 1024 * 1024

_DEFAULT_EXCLUDED_DIRECTORIES = frozenset(
    {
        ".agent-supervisor",
        ".cache",
        ".codex",
        ".eggs",
        ".git",
        ".gradle",
        ".hypothesis",
        ".mypy_cache",
        ".nox",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        ".venv",
        "__pycache__",
        "bower_components",
        "build",
        "coverage",
        "dist",
        "generated",
        "htmlcov",
        "node_modules",
        "site-packages",
        "supervisor-state",
        "target",
        "vendor",
        "venv",
        "worktrees",
    }
)
_CREDENTIAL_FILENAMES = frozenset(
    {
        ".env",
        ".netrc",
        "_netrc",
        "credentials",
        "credentials.json",
        "credentials.yaml",
        "credentials.yml",
        "id_dsa",
        "id_ecdsa",
        "id_ed25519",
        "id_rsa",
        "secrets.json",
        "secrets.yaml",
        "secrets.yml",
    }
)
_CREDENTIAL_SUFFIXES = frozenset(
    {".der", ".jks", ".key", ".keystore", ".p12", ".pem", ".pfx", ".pkcs12"}
)
_BINARY_SUFFIXES = frozenset(
    {
        ".7z",
        ".a",
        ".avi",
        ".bin",
        ".bmp",
        ".bz2",
        ".class",
        ".db",
        ".dll",
        ".dylib",
        ".duckdb",
        ".eot",
        ".exe",
        ".flac",
        ".gif",
        ".gz",
        ".ico",
        ".jar",
        ".jpeg",
        ".jpg",
        ".lockb",
        ".mov",
        ".mp3",
        ".mp4",
        ".o",
        ".otf",
        ".parquet",
        ".pdf",
        ".png",
        ".pyc",
        ".pyo",
        ".so",
        ".sqlite",
        ".sqlite3",
        ".tar",
        ".tiff",
        ".ttf",
        ".wav",
        ".webm",
        ".webp",
        ".woff",
        ".woff2",
        ".xz",
        ".zip",
    }
)
_DOCUMENT_SUFFIXES = frozenset(
    {".adoc", ".md", ".mdx", ".rst", ".txt"}
)
_POLICY_NAMES = frozenset(
    {
        "agents.md",
        "code_of_conduct.md",
        "codeowners",
        "contributing.md",
        "license",
        "license.md",
        "notice",
        "security.md",
    }
)
_BUILD_NAMES = frozenset(
    {
        "build.gradle",
        "build.gradle.kts",
        "cargo.lock",
        "cargo.toml",
        "cmakelists.txt",
        "composer.json",
        "dockerfile",
        "gemfile",
        "go.mod",
        "go.sum",
        "gradle.properties",
        "makefile",
        "meson.build",
        "package-lock.json",
        "package.json",
        "pnpm-lock.yaml",
        "poetry.lock",
        "pom.xml",
        "pyproject.toml",
        "requirements.txt",
        "setup.cfg",
        "setup.py",
        "tox.ini",
        "yarn.lock",
    }
)
_LANGUAGES = MappingProxyType(
    {
        ".c": "C",
        ".cc": "C++",
        ".cpp": "C++",
        ".cs": "C#",
        ".css": "CSS",
        ".go": "Go",
        ".h": "C/C++ header",
        ".hpp": "C++ header",
        ".html": "HTML",
        ".java": "Java",
        ".js": "JavaScript",
        ".json": "JSON",
        ".jsx": "JavaScript",
        ".kt": "Kotlin",
        ".kts": "Kotlin",
        ".lua": "Lua",
        ".md": "Markdown",
        ".php": "PHP",
        ".proto": "Protocol Buffers",
        ".py": "Python",
        ".rb": "Ruby",
        ".rs": "Rust",
        ".scss": "SCSS",
        ".sh": "Shell",
        ".sql": "SQL",
        ".swift": "Swift",
        ".toml": "TOML",
        ".ts": "TypeScript",
        ".tsx": "TypeScript",
        ".xml": "XML",
        ".yaml": "YAML",
        ".yml": "YAML",
    }
)
_SECRET_PATTERNS = (
    re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(rb"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(rb"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    re.compile(rb"\bsk-[A-Za-z0-9_-]{20,}\b"),
    re.compile(rb"\bxox[baprs]-[A-Za-z0-9-]{16,}\b"),
    re.compile(
        rb"(?i)\b(?:api[_-]?key|password|private[_-]?key|secret|token)"
        rb"\s*[:=]\s*[\"'][^\"'\r\n]{12,}[\"']"
    ),
)
_SAFE_REASON_RE = re.compile(r"[^a-zA-Z0-9_.:@/+ -]")


class PromptDirectoryScanError(RuntimeError):
    """Base error for an unsafe, unstable, or malformed directory scan."""


class RepositoryAllowlistError(PromptDirectoryScanError, ValueError):
    """The requested repository is not bound to the explicit allowlist."""


class DirectoryResolutionError(PromptDirectoryScanError, ValueError):
    """The repository, directory, or output target does not resolve safely."""


class NestedRepositoryError(PromptDirectoryScanError, ValueError):
    """A nested Git repository makes the requested source closure ambiguous."""


class ScanSymlinkError(PromptDirectoryScanError, ValueError):
    """A selected symlink is forbidden or escapes the selected root."""


class ScanBudgetError(PromptDirectoryScanError, ValueError):
    """The scan cannot complete within a declared hard resource bound."""


class SecretLeakageError(PromptDirectoryScanError):
    """Secret-like bytes occurred in a policy-admitted input."""


class UnstableDirectoryScanError(PromptDirectoryScanError):
    """A root or admitted byte changed while the scan was being constructed."""


class ApproximateEvidenceAuthorityError(PromptDirectoryScanError, ValueError):
    """An optional/approximate analyzer attempted to claim authority."""


@dataclass(frozen=True)
class RepositoryAllowlist:
    """Canonical allowlist of exact resolved Git worktree roots."""

    repository_roots: tuple[str, ...]
    allowlist_cid: str

    def __post_init__(self) -> None:
        roots = tuple(sorted({_canonical_existing_directory(item) for item in self.repository_roots}))
        if not roots:
            raise RepositoryAllowlistError(
                "repository allowlist must contain at least one exact root"
            )
        expected = _cid_for(
            {
                "schema": REPOSITORY_ALLOWLIST_SCHEMA,
                "repository_roots": list(roots),
            }
        )
        if str(self.allowlist_cid) != expected:
            raise RepositoryAllowlistError(
                "repository allowlist identity does not match its resolved roots"
            )
        object.__setattr__(self, "repository_roots", roots)
        object.__setattr__(self, "allowlist_cid", expected)

    @classmethod
    def from_roots(
        cls, roots: Sequence[str | os.PathLike[str]]
    ) -> "RepositoryAllowlist":
        normalized = tuple(
            sorted({_canonical_existing_directory(item) for item in roots})
        )
        return cls(
            normalized,
            _cid_for(
                {
                    "schema": REPOSITORY_ALLOWLIST_SCHEMA,
                    "repository_roots": list(normalized),
                }
            ),
        )


@dataclass(frozen=True)
class ScanArtifact:
    """A bounded, body-free evidence manifest and its storage handle."""

    kind: str
    artifact_cid: str
    artifact_handle: str
    size_bytes: int
    payload: Mapping[str, Any] = field(repr=False)


@dataclass(frozen=True)
class OptionalAnalysisContext:
    """Body-free identity and summary context given to an optional adapter."""

    request_cid: str
    repository_root_cid: str
    dirty_worktree_root: str
    scanner_policy_cid: str
    program_root: str
    ast_root: str
    configuration_root: str
    included_paths: tuple[str, ...]
    category_counts: Mapping[str, int]
    max_summary_bytes: int


@dataclass(frozen=True)
class OptionalAnalysisResult:
    """Strict advisory projection returned by an optional scan adapter."""

    status: str
    summary: str
    artifact_cid: str = ""
    repository_paths: tuple[str, ...] = ()
    claim_keys: tuple[str, ...] = ()
    authority: EvidenceAuthority = EvidenceAuthority.SCAN_ADVISORY
    repository_root_cid: str = ""
    dirty_worktree_root: str = ""
    scanner_policy_cid: str = ""


class OptionalAnalysisAdapter(Protocol):
    def analyze(self, context: OptionalAnalysisContext) -> OptionalAnalysisResult:
        """Return one advisory, body-free result."""


@dataclass(frozen=True)
class DirectoryScanDetails:
    """Non-durable local details for callers that need artifact persistence."""

    receipt: DirectoryScanReceipt
    artifacts: tuple[ScanArtifact, ...]
    program_behavior: "ProgramBehavior"
    configuration_root: str
    stability_checks: tuple[str, ...]
    optional_analysis_status: str

    @property
    def scan_cid(self) -> str:
        return self.receipt.scan_cid


@dataclass(frozen=True)
class _GitEntry:
    mode: str
    object_id: str


@dataclass(frozen=True)
class _Decision:
    path: str
    disposition: str
    reason: str
    redactions: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "disposition": self.disposition,
            "reason": self.reason,
            "redactions": list(self.redactions),
        }


@dataclass(frozen=True)
class _RootWitness:
    repository_signature: tuple[int, int, int]
    directory_signature: tuple[int, int, int]
    git_top_level: str
    git_directory: str


def _cid_for(value: Any) -> str:
    return "sha256:" + hashlib.sha256(
        canonical_prompt_workflow_bytes(value)
    ).hexdigest()


def _canonical_existing_directory(value: str | os.PathLike[str]) -> str:
    path = Path(value)
    if not path.is_absolute():
        raise RepositoryAllowlistError("repository allowlist roots must be absolute")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise RepositoryAllowlistError(
            "repository allowlist root is unavailable"
        ) from exc
    if resolved != path or not resolved.is_dir() or resolved == Path("/"):
        raise RepositoryAllowlistError(
            "repository allowlist roots must be canonical non-root directories"
        )
    return str(resolved)


def repository_root_cid(repository_root: str | os.PathLike[str]) -> str:
    """Return the canonical identity for one exact resolved repository root."""

    root = _canonical_existing_directory(repository_root)
    return _cid_for({"schema": REPOSITORY_ROOT_SCHEMA, "repository_root": root})


def repository_allowlist_cid(
    repository_roots: Sequence[str | os.PathLike[str]],
) -> str:
    """Return the canonical identity for an exact repository allowlist."""

    return RepositoryAllowlist.from_roots(repository_roots).allowlist_cid


def build_repository_allowlist(
    repository_roots: Sequence[str | os.PathLike[str]],
) -> RepositoryAllowlist:
    return RepositoryAllowlist.from_roots(repository_roots)


def _remaining_seconds(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise ScanBudgetError("directory scan exceeded max_latency_ms")
    return max(0.001, remaining)


def _run_git(
    root: Path,
    arguments: Sequence[str],
    *,
    deadline: float,
    input_bytes: bytes | None = None,
    allow_failure: bool = False,
) -> bytes:
    try:
        result = subprocess.run(
            (
                "git",
                "-c",
                "core.quotepath=false",
                "-C",
                str(root),
                *arguments,
            ),
            input=input_bytes,
            stdin=subprocess.PIPE if input_bytes is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=_remaining_seconds(deadline),
        )
    except subprocess.TimeoutExpired as exc:
        raise ScanBudgetError("Git inspection exceeded max_latency_ms") from exc
    except OSError as exc:
        raise DirectoryResolutionError("Git executable is unavailable") from exc
    if result.returncode and not allow_failure:
        raise DirectoryResolutionError(
            f"Git operation {arguments[0]!r} failed"
        )
    return result.stdout if not result.returncode else b""


def _git_entries(
    root: Path, *, deadline: float
) -> tuple[dict[str, _GitEntry], dict[str, _GitEntry]]:
    head: dict[str, _GitEntry] = {}
    for record in _run_git(
        root, ("ls-tree", "-rz", "--full-tree", "HEAD"), deadline=deadline,
        allow_failure=True,
    ).split(b"\0"):
        if not record:
            continue
        try:
            metadata, raw_path = record.split(b"\t", 1)
            mode, kind, object_id = metadata.decode("ascii").split()
            path = raw_path.decode("utf-8")
        except (ValueError, UnicodeDecodeError) as exc:
            raise DirectoryResolutionError(
                "HEAD contains an undecodable entry"
            ) from exc
        if kind != "blob" or mode == "160000":
            raise NestedRepositoryError(
                f"nested repository or submodule is forbidden at {path!r}"
            )
        head[path] = _GitEntry(mode, object_id)

    index: dict[str, _GitEntry] = {}
    for record in _run_git(
        root, ("ls-files", "--stage", "-z"), deadline=deadline
    ).split(b"\0"):
        if not record:
            continue
        try:
            metadata, raw_path = record.split(b"\t", 1)
            mode, object_id, stage = metadata.decode("ascii").split()
            path = raw_path.decode("utf-8")
        except (ValueError, UnicodeDecodeError) as exc:
            raise DirectoryResolutionError(
                "Git index contains an undecodable entry"
            ) from exc
        if stage != "0":
            raise DirectoryResolutionError(
                f"unmerged Git index entry is forbidden at {path!r}"
            )
        if mode == "160000":
            raise NestedRepositoryError(
                f"nested repository or submodule is forbidden at {path!r}"
            )
        index[path] = _GitEntry(mode, object_id)
    return head, index


def _in_scope(path: str, scope: str) -> bool:
    return scope == "." or path == scope or path.startswith(scope + "/")


def _default_directory_reason(
    path: str, *, exclude_generated_state: bool = True
) -> str:
    parts = PurePosixPath(path).parts
    for part in parts:
        folded = part.casefold()
        if folded in _DEFAULT_EXCLUDED_DIRECTORIES:
            if folded in {
                ".agent-supervisor",
                ".codex",
                "supervisor-state",
                "worktrees",
            }:
                return "supervisor_generated_state"
            if folded == ".git":
                return "git_metadata"
            if not exclude_generated_state:
                continue
            if folded in {"vendor", "node_modules", "bower_components", "site-packages"}:
                return "vendor_tree"
            if folded in {"build", "dist", "generated", "target"}:
                return "generated_tree"
            return "cache_tree"
    return ""


def _credential_reason(path: str) -> str:
    name = PurePosixPath(path).name.casefold()
    suffix = PurePosixPath(name).suffix
    if (
        name in _CREDENTIAL_FILENAMES
        or name.startswith(".env.")
        or suffix in _CREDENTIAL_SUFFIXES
    ):
        return "credential_or_key_material"
    return ""


def _matches(path: str, patterns: Sequence[str]) -> bool:
    name = PurePosixPath(path).name
    for pattern in patterns:
        if (
            fnmatch.fnmatchcase(path, pattern)
            or fnmatch.fnmatchcase(name, pattern)
            or (
                pattern.startswith("**/")
                and fnmatch.fnmatchcase(path, pattern[3:])
            )
        ):
            return True
    return False


def _safe_reason(value: str) -> str:
    rendered = _SAFE_REASON_RE.sub("_", str(value))[:120]
    return rendered or "unknown"


def _root_witness(
    root: Path, directory: Path, *, deadline: float
) -> _RootWitness:
    try:
        root_stat = root.stat()
        directory_stat = directory.stat()
    except OSError as exc:
        raise DirectoryResolutionError("scan roots are unavailable") from exc
    top = _run_git(
        root, ("rev-parse", "--show-toplevel"), deadline=deadline
    ).decode("utf-8", "strict").strip()
    git_directory = _run_git(
        root, ("rev-parse", "--absolute-git-dir"), deadline=deadline
    ).decode("utf-8", "strict").strip()
    return _RootWitness(
        (root_stat.st_dev, root_stat.st_ino, root_stat.st_mode),
        (directory_stat.st_dev, directory_stat.st_ino, directory_stat.st_mode),
        top,
        str(Path(git_directory).resolve(strict=True)),
    )


def _resolve_request(
    request: PromptWorkflowRequest,
    allowlist: RepositoryAllowlist,
    *,
    deadline: float,
) -> tuple[Path, Path, str, _RootWitness, tuple[str, ...]]:
    root = Path(request.repository_root)
    directory = Path(request.directory)
    try:
        resolved_root = root.resolve(strict=True)
        resolved_directory = directory.resolve(strict=True)
    except OSError as exc:
        raise DirectoryResolutionError(
            "repository root or requested directory is unavailable"
        ) from exc
    if resolved_root != root or resolved_directory != directory:
        raise DirectoryResolutionError(
            "repository root and directory must not traverse symlinks"
        )
    if not resolved_directory.is_dir():
        raise DirectoryResolutionError("requested directory must be a directory")
    try:
        common = Path(os.path.commonpath((str(root), str(directory))))
    except ValueError as exc:
        raise DirectoryResolutionError(
            "requested directory is outside repository root"
        ) from exc
    if common != root:
        raise DirectoryResolutionError(
            "requested directory is outside repository root"
        )
    if str(root) not in allowlist.repository_roots:
        raise RepositoryAllowlistError(
            "requested repository is not in the explicit repository allowlist"
        )
    if request.allowlist_cid != allowlist.allowlist_cid:
        raise RepositoryAllowlistError(
            "request allowlist identity does not match explicit allowlist"
        )
    expected_root_cid = repository_root_cid(root)
    if request.repository_root_cid != expected_root_cid:
        raise RepositoryAllowlistError(
            "request repository root identity does not match resolved repository root"
        )
    witness = _root_witness(root, directory, deadline=deadline)
    if Path(witness.git_top_level).resolve(strict=True) != root:
        raise DirectoryResolutionError(
            "repository_root must name the exact Git worktree top level"
        )
    scope = directory.relative_to(root).as_posix()
    if scope == ".":
        scope = "."
    output_exclusions = _validate_output_paths(request, root, scope)
    return root, directory, scope, witness, output_exclusions


def _validate_output_paths(
    request: PromptWorkflowRequest, root: Path, scope: str
) -> tuple[str, ...]:
    policy = request.output_policy
    output_root = Path(policy.output_root)
    resolved_output_root = output_root.resolve(strict=False)
    if resolved_output_root != output_root:
        raise DirectoryResolutionError(
            "output_root traverses a symlink or non-canonical ancestor"
        )
    for allowed in policy.allowed_output_roots:
        allowed_path = Path(allowed)
        if allowed_path.resolve(strict=False) != allowed_path:
            raise DirectoryResolutionError(
                "allowed output root traverses a symlink"
            )
    results: list[str] = []
    for relative in (policy.markdown_path, policy.duckdb_path):
        if not relative:
            continue
        target = output_root.joinpath(*PurePosixPath(relative).parts)
        resolved = target.resolve(strict=False)
        try:
            if (
                Path(os.path.commonpath((str(output_root), str(resolved))))
                != output_root
            ):
                raise DirectoryResolutionError(
                    "output path escapes its declared output_root"
                )
        except ValueError as exc:
            raise DirectoryResolutionError(
                "output path escapes its declared output_root"
            ) from exc
        try:
            repository_relative = target.relative_to(root).as_posix()
        except ValueError:
            continue
        if _in_scope(repository_relative, scope):
            results.append(repository_relative)
    return tuple(sorted(set(results)))


def _worktree_paths(
    root: Path,
    directory: Path,
    scope: str,
    *,
    reject_nested: bool,
    exclude_generated_state: bool,
    deadline: float,
) -> tuple[set[str], set[str]]:
    files: set[str] = set()
    excluded_directories: set[str] = set()

    def traversal_error(error: OSError) -> None:
        raise DirectoryResolutionError(
            "requested directory contains an unreadable path"
        ) from error

    for current, names, filenames in os.walk(
        directory, topdown=True, followlinks=False, onerror=traversal_error
    ):
        _remaining_seconds(deadline)
        current_path = Path(current)
        kept: list[str] = []
        for name in sorted(names):
            child = current_path / name
            relative = child.relative_to(root).as_posix()
            default_reason = _default_directory_reason(
                relative,
                exclude_generated_state=exclude_generated_state,
            )
            if name == ".git":
                is_root_git = current_path == root
                if (
                    not is_root_git
                    and reject_nested
                    and not _default_directory_reason(
                        current_path.relative_to(root).as_posix(),
                        exclude_generated_state=exclude_generated_state,
                    )
                ):
                    raise NestedRepositoryError(
                        f"nested repository is forbidden at {current_path.relative_to(root).as_posix()!r}"
                    )
                excluded_directories.add(relative)
                continue
            if default_reason:
                excluded_directories.add(relative)
                continue
            try:
                child_stat = child.lstat()
            except OSError as exc:
                raise DirectoryResolutionError(
                    f"directory entry is unreadable: {relative!r}"
                ) from exc
            if stat.S_ISLNK(child_stat.st_mode):
                files.add(relative)
            elif stat.S_ISDIR(child_stat.st_mode):
                kept.append(name)
            else:
                raise DirectoryResolutionError(
                    f"special directory entry is forbidden: {relative!r}"
                )
        names[:] = kept
        for name in sorted(filenames):
            relative = (current_path / name).relative_to(root).as_posix()
            if (
                name == ".git"
                and current_path != root
                and reject_nested
                and not _default_directory_reason(
                    current_path.relative_to(root).as_posix(),
                    exclude_generated_state=exclude_generated_state,
                )
            ):
                raise NestedRepositoryError(
                    f"nested repository is forbidden at {current_path.relative_to(root).as_posix()!r}"
                )
            files.add(relative)
    if scope == ".":
        excluded_directories.add(".git")
    return files, excluded_directories


def _ignored_paths(root: Path, scope: str, *, deadline: float) -> set[str]:
    pathspec = "." if scope == "." else scope
    output = _run_git(
        root,
        (
            "ls-files",
            "--others",
            "--ignored",
            "--exclude-standard",
            "-z",
            "--",
            pathspec,
        ),
        deadline=deadline,
    )
    try:
        return {
            item.decode("utf-8")
            for item in output.split(b"\0")
            if item
        }
    except UnicodeDecodeError as exc:
        raise DirectoryResolutionError(
            "ignored worktree path is not UTF-8"
        ) from exc


def _stable_file_bytes(path: Path, relative: str, maximum: int) -> bytes:
    try:
        before = path.lstat()
    except OSError as exc:
        raise UnstableDirectoryScanError(
            f"admitted path became unreadable: {relative!r}"
        ) from exc
    if stat.S_ISLNK(before.st_mode):
        try:
            target = os.readlink(path)
            after = path.lstat()
        except OSError as exc:
            raise UnstableDirectoryScanError(
                f"admitted symlink became unreadable: {relative!r}"
            ) from exc
        if _stat_signature(before) != _stat_signature(after):
            raise UnstableDirectoryScanError(
                f"admitted symlink changed while scanning: {relative!r}"
            )
        data = os.fsencode(target)
        if len(data) > maximum:
            raise ScanBudgetError(
                f"admitted symlink exceeds max_file_bytes: {relative!r}"
            )
        return data
    if not stat.S_ISREG(before.st_mode):
        raise DirectoryResolutionError(
            f"special file is forbidden: {relative!r}"
        )
    if before.st_size > maximum:
        raise ScanBudgetError(
            f"admitted file exceeds max_file_bytes: {relative!r}"
        )
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
        try:
            opened = os.fstat(descriptor)
            if _stat_signature(before) != _stat_signature(opened):
                raise UnstableDirectoryScanError(
                    f"admitted file changed before scanning: {relative!r}"
                )
            chunks: list[bytes] = []
            total = 0
            while True:
                chunk = os.read(descriptor, min(1024 * 1024, maximum + 1 - total))
                if not chunk:
                    break
                chunks.append(chunk)
                total += len(chunk)
                if total > maximum:
                    raise ScanBudgetError(
                        f"admitted file exceeds max_file_bytes: {relative!r}"
                    )
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        final = path.lstat()
    except PromptDirectoryScanError:
        raise
    except OSError as exc:
        raise UnstableDirectoryScanError(
            f"admitted file became unreadable: {relative!r}"
        ) from exc
    if not (
        _stat_signature(before)
        == _stat_signature(after)
        == _stat_signature(final)
    ):
        raise UnstableDirectoryScanError(
            f"admitted file changed while scanning: {relative!r}"
        )
    return b"".join(chunks)


def _stat_signature(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
    )


def _contains_secret(data: bytes) -> bool:
    return any(pattern.search(data) for pattern in _SECRET_PATTERNS)


def _looks_binary(path: str, data: bytes | None = None) -> bool:
    if PurePosixPath(path).suffix.casefold() in _BINARY_SUFFIXES:
        return True
    if data is None:
        return False
    sample = data[:8192]
    if b"\x00" in sample:
        return True
    try:
        sample.decode("utf-8")
    except UnicodeDecodeError:
        return True
    return False


def _artifact(
    kind: str,
    payload: Mapping[str, Any],
    *,
    artifact_store: "BoundedArtifactStore | None",
) -> ScanArtifact:
    try:
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise PromptDirectoryScanError(
            "scan evidence artifact is not canonical JSON"
        ) from exc
    if len(encoded) > MAX_EVIDENCE_ARTIFACT_BYTES:
        raise ScanBudgetError(
            "scan evidence artifact exceeds the 16777216-byte hard bound"
        )
    digest = "sha256:" + hashlib.sha256(encoded).hexdigest()
    handle = f"blob:{digest}"
    if artifact_store is not None:
        reference = artifact_store.put_blob(
            encoded,
            kind=f"prompt-directory-scan:{kind}",
            media_type="application/json",
        )
        if reference.digest != digest:
            raise PromptDirectoryScanError(
                "artifact store returned a mismatched content digest"
            )
        handle = reference.artifact_id
    return ScanArtifact(
        kind=kind,
        artifact_cid=digest,
        artifact_handle=handle,
        size_bytes=len(encoded),
        payload=MappingProxyType(dict(payload)),
    )


def _category_paths(paths: Sequence[str], maximum: int = 256) -> tuple[str, ...]:
    return tuple(sorted(set(paths))[:maximum])


def _evidence_record(
    *,
    key: str,
    kind: str,
    summary: str,
    artifact: ScanArtifact,
    paths: Sequence[str] = (),
    claims: Sequence[str] = (),
    extra_provenance: Mapping[str, Any] | None = None,
) -> PromptEvidenceRecord:
    selected_paths = _category_paths(paths)
    provenance: dict[str, Any] = {
        "artifact_handle": artifact.artifact_handle,
        "artifact_size_bytes": artifact.size_bytes,
        "body_handling": "content-address-only",
        "path_count": len(set(paths)),
        "receipt_path_count": len(selected_paths),
        "producer": f"prompt-directory-scanner@{PROMPT_DIRECTORY_SCANNER_VERSION}",
    }
    if extra_provenance:
        provenance.update(extra_provenance)
    return PromptEvidenceRecord(
        evidence_key=key,
        source_kind=kind,
        artifact_cid=artifact.artifact_cid,
        summary=summary,
        repository_paths=selected_paths,
        claim_keys=tuple(claims),
        authority=EvidenceAuthority.SCAN_ADVISORY,
        provenance=provenance,
        status=RecordStatus.ADMITTED,
    )


_CONFIG_NAMES = frozenset(
    {
        ".editorconfig",
        ".flake8",
        ".gitattributes",
        ".gitignore",
        ".pre-commit-config.yaml",
        "pytest.ini",
        "ruff.toml",
        "setup.cfg",
        "tox.ini",
    }
)
_CONFIG_SUFFIXES = frozenset({".toml", ".ini", ".cfg", ".conf", ".config"})
_SCHEMA_SUFFIXES = frozenset(
    {".jsonschema", ".schema", ".proto", ".avsc", ".graphql", ".xsd"}
)


def _classify(
    behavior: "ProgramBehavior", included_paths: Sequence[str], max_symbols: int
) -> tuple[dict[str, Mapping[str, Any]], list[str], dict[str, int]]:
    included = tuple(sorted(included_paths))
    languages: Counter[str] = Counter()
    language_paths: dict[str, list[str]] = {}
    builds: list[str] = []
    tests: list[str] = []
    documents: list[str] = []
    policies: list[str] = []
    configs: list[str] = []
    schemas: list[str] = []
    for path in included:
        pure = PurePosixPath(path)
        name = pure.name.casefold()
        parts = {part.casefold() for part in pure.parts[:-1]}
        suffix = pure.suffix.casefold()
        language = _LANGUAGES.get(suffix)
        if language:
            languages[language] += 1
            language_paths.setdefault(language, []).append(path)
        if (
            name in _BUILD_NAMES
            or name.startswith("requirements")
            and suffix == ".txt"
        ):
            builds.append(path)
        if (
            "test" in parts
            or "tests" in parts
            or name.startswith("test_")
            or name.endswith("_test.py")
            or name.endswith(".spec.js")
            or name.endswith(".test.ts")
        ):
            tests.append(path)
        if suffix in _DOCUMENT_SUFFIXES:
            documents.append(path)
        if (
            name in _POLICY_NAMES
            or "policy" in name
            or name.endswith(".todo.md")
            or name.endswith(".objectives.md")
        ):
            policies.append(path)
        if (
            "config" in parts
            or "configs" in parts
            or name in _CONFIG_NAMES
            or suffix in _CONFIG_SUFFIXES
            or (suffix in {".yaml", ".yml", ".json"} and "config" in name)
        ):
            configs.append(path)
        if (
            "schema" in parts
            or "schemas" in parts
            or name.endswith(".schema.json")
            or suffix in _SCHEMA_SUFFIXES
            or pure.stem.endswith("_schema")
        ):
            schemas.append(path)

    from ..core.program_behavior import ProgramObservationKind

    all_interface_observations = [
        item
        for item in behavior.analysis.observations
        if item.kind is ProgramObservationKind.INTERFACE
    ]
    interface_observations = all_interface_observations[:max_symbols]
    symbol_observations = [
        item
        for item in behavior.analysis.observations
        if item.kind is ProgramObservationKind.SYMBOL
    ]
    symbol_truncated = len(symbol_observations) > max_symbols
    selected_symbols = symbol_observations[:max_symbols]
    categories: dict[str, Mapping[str, Any]] = {
        "languages": {
            "counts": dict(sorted(languages.items())),
            "paths": {
                key: sorted(value) for key, value in sorted(language_paths.items())
            },
        },
        "build": {"paths": sorted(builds), "count": len(builds)},
        "interfaces": {
            "count": len(interface_observations),
            "total_count": len(all_interface_observations),
            "truncated": len(all_interface_observations) > max_symbols,
            "records": [
                {
                    "path": item.path,
                    "subject": item.subject,
                    "relationship": item.relationship,
                    "target": item.target,
                    "ast_record_id": item.ast_record_id,
                }
                for item in interface_observations
            ],
        },
        "symbols": {
            "count": len(selected_symbols),
            "total_count": len(symbol_observations),
            "truncated": symbol_truncated,
            "records": [
                {
                    "path": item.path,
                    "symbol": item.subject,
                    "symbol_hash": item.symbol_hash,
                    "ast_record_id": item.ast_record_id,
                }
                for item in selected_symbols
            ],
        },
        "tests": {"paths": sorted(tests), "count": len(tests)},
        "documents": {"paths": sorted(documents), "count": len(documents)},
        "policies": {"paths": sorted(policies), "count": len(policies)},
        "config": {"paths": sorted(configs), "count": len(configs)},
        "schema": {"paths": sorted(schemas), "count": len(schemas)},
    }
    truncations = []
    if symbol_truncated:
        truncations.append(
            "symbol_summary:max_symbols:"
            f"{len(selected_symbols)}-of-{len(symbol_observations)}"
        )
    if len(all_interface_observations) > max_symbols:
        truncations.append(
            "interface_summary:max_symbols:"
            f"{len(interface_observations)}-of-{len(all_interface_observations)}"
        )
    counts = {
        "languages": len(languages),
        "build_files": len(builds),
        "interfaces": len(interface_observations),
        "symbols": len(selected_symbols),
        "tests": len(tests),
        "documents": len(documents),
        "policies": len(policies),
        "config": len(configs),
        "schema": len(schemas),
    }
    return categories, truncations, counts


def _category_summary(name: str, payload: Mapping[str, Any]) -> str:
    if name == "languages":
        counts = payload["counts"]
        rendered = ", ".join(f"{key}={counts[key]}" for key in sorted(counts))
        return f"Language summary: {rendered or 'no recognized language files'}."
    if name == "symbols":
        return (
            f"Symbol summary contains {payload['count']} of "
            f"{payload['total_count']} exact AST-defined symbols."
        )
    return f"{name.capitalize()} summary contains {payload.get('count', 0)} records."


def _optional_result(
    adapter: OptionalAnalysisAdapter | Callable[[OptionalAnalysisContext], Any] | None,
    context: OptionalAnalysisContext,
) -> tuple[OptionalAnalysisResult, str]:
    if adapter is None:
        return (
            OptionalAnalysisResult(
                status="not_requested",
                summary="Optional analysis was not requested; exact local scanning completed.",
            ),
            "not_requested",
        )
    try:
        operation = getattr(adapter, "analyze", adapter)
        result = operation(context)
        if inspect.isawaitable(result):
            return (
                OptionalAnalysisResult(
                    status="degraded",
                    summary="Optional analysis returned an unsupported asynchronous result.",
                ),
                "async_result_unsupported",
            )
        if isinstance(result, OptionalAnalysisResult):
            parsed = result
        elif isinstance(result, Mapping):
            allowed = {
                "status",
                "summary",
                "artifact_cid",
                "repository_paths",
                "claim_keys",
                "authority",
                "repository_root_cid",
                "dirty_worktree_root",
                "scanner_policy_cid",
            }
            if set(result).difference(allowed):
                raise ValueError("optional analysis result contains unknown fields")
            parsed = OptionalAnalysisResult(
                status=str(result.get("status", "available")),
                summary=str(result.get("summary", "Optional advisory analysis completed.")),
                artifact_cid=str(result.get("artifact_cid", "")),
                repository_paths=tuple(result.get("repository_paths", ())),
                claim_keys=tuple(result.get("claim_keys", ())),
                authority=EvidenceAuthority(
                    result.get("authority", EvidenceAuthority.SCAN_ADVISORY)
                ),
                repository_root_cid=str(result.get("repository_root_cid", "")),
                dirty_worktree_root=str(result.get("dirty_worktree_root", "")),
                scanner_policy_cid=str(result.get("scanner_policy_cid", "")),
            )
        else:
            raise TypeError("optional analysis result has unsupported type")
    except Exception as exc:  # optional analysis must not abort exact local work
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        reason = _safe_reason(type(exc).__name__).casefold()
        return (
            OptionalAnalysisResult(
                status="degraded",
                summary=f"Optional analysis degraded safely ({reason}).",
            ),
            f"invocation_failed:{reason}",
        )

    if parsed.authority is not EvidenceAuthority.SCAN_ADVISORY:
        return (
            OptionalAnalysisResult(
                status="rejected",
                summary="Optional analysis authority claim was rejected.",
            ),
            "authority_claim_rejected",
        )
    bindings = (
        ("repository_root_cid", parsed.repository_root_cid, context.repository_root_cid),
        ("dirty_worktree_root", parsed.dirty_worktree_root, context.dirty_worktree_root),
        ("scanner_policy_cid", parsed.scanner_policy_cid, context.scanner_policy_cid),
    )
    if any(claimed and claimed != expected for _, claimed, expected in bindings):
        return (
            OptionalAnalysisResult(
                status="rejected",
                summary="Optional analysis identity binding was rejected.",
            ),
            "identity_mismatch",
        )
    if len(parsed.summary.encode("utf-8")) > context.max_summary_bytes:
        return (
            OptionalAnalysisResult(
                status="degraded",
                summary="Optional analysis summary exceeded its byte bound.",
            ),
            "summary_too_large",
        )
    try:
        paths = tuple(str(item) for item in parsed.repository_paths)
        claims = tuple(str(item) for item in parsed.claim_keys)
        if len(paths) > 1_024 or len(claims) > 1_024:
            raise ValueError("optional analysis projection has too many items")
        if len(paths) != len(set(paths)) or len(claims) != len(set(claims)):
            raise ValueError("optional analysis projection contains duplicates")
        included = set(context.included_paths)
        for path in paths:
            candidate = PurePosixPath(path)
            if (
                not path
                or candidate.is_absolute()
                or ".." in candidate.parts
                or candidate.as_posix() != path
                or path not in included
            ):
                raise ValueError("optional analysis path is outside scan closure")
        if any(
            not claim
            or len(claim.encode("utf-8")) > 1_024
            or "\x00" in claim
            for claim in claims
        ):
            raise ValueError("optional analysis claim is malformed")
        summary_bytes = parsed.summary.encode("utf-8")
        if (
            not parsed.summary
            or parsed.summary != parsed.summary.strip()
            or b"\x00" in summary_bytes
            or _contains_secret(summary_bytes)
        ):
            raise SecretLeakageError(
                "optional analysis summary failed secret screening"
            )
        if (
            not parsed.status
            or parsed.status != parsed.status.strip()
            or len(parsed.status.encode("utf-8")) > 128
        ):
            raise ValueError("optional analysis status is malformed")
        if (
            parsed.artifact_cid
            and not re.fullmatch(
                r"(?:sha256:[0-9a-f]{64}|b[a-z2-7]+)",
                parsed.artifact_cid,
            )
        ):
            raise ValueError("optional artifact identity is malformed")
    except Exception as exc:
        reason = _safe_reason(type(exc).__name__).casefold()
        return (
            OptionalAnalysisResult(
                status="degraded",
                summary=f"Optional analysis projection was rejected safely ({reason}).",
            ),
            f"projection_rejected:{reason}",
        )
    return parsed, parsed.status


def _coerce_allowlist(
    value: RepositoryAllowlist | Sequence[str | os.PathLike[str]],
) -> RepositoryAllowlist:
    if isinstance(value, RepositoryAllowlist):
        return value
    if isinstance(value, (str, bytes, os.PathLike)):
        raise RepositoryAllowlistError(
            "repository_allowlist must be an explicit sequence of roots"
        )
    return RepositoryAllowlist.from_roots(value)


def _verify_root_witness(
    root: Path,
    directory: Path,
    expected: _RootWitness,
    *,
    deadline: float,
) -> None:
    current = _root_witness(root, directory, deadline=deadline)
    if current != expected:
        raise UnstableDirectoryScanError(
            "repository or directory root changed during scan"
        )


def scan_prompt_directory_detailed(
    request: PromptWorkflowRequest,
    *,
    repository_allowlist: RepositoryAllowlist
    | Sequence[str | os.PathLike[str]],
    artifact_store: "BoundedArtifactStore | None" = None,
    optional_analysis: OptionalAnalysisAdapter
    | Callable[[OptionalAnalysisContext], Any]
    | None = None,
    previous: DirectoryScanDetails | None = None,
    clock_ms: Callable[[], int] | None = None,
) -> DirectoryScanDetails:
    """Build a stable receipt plus local body-free evidence manifests.

    ``previous`` enables the existing program-behavior AST reuse path.  It is
    only a performance hint: every current byte and every returned root is
    still verified.
    """

    if not isinstance(request, PromptWorkflowRequest):
        raise TypeError("request must be PromptWorkflowRequest")
    allowlist = _coerce_allowlist(repository_allowlist)
    budget = request.budget
    started_at_ms = int((clock_ms or (lambda: time.time_ns() // 1_000_000))())
    deadline = time.monotonic() + budget.max_latency_ms / 1000.0
    root, directory, scope, witness, output_exclusions = _resolve_request(
        request, allowlist, deadline=deadline
    )
    head, index = _git_entries(root, deadline=deadline)
    worktree, excluded_directories = _worktree_paths(
        root,
        directory,
        scope,
        reject_nested=request.scan_policy.reject_nested_repositories,
        exclude_generated_state=request.scan_policy.exclude_generated_state,
        deadline=deadline,
    )
    ignored = _ignored_paths(root, scope, deadline=deadline)
    tracked = set(head) | set(index)
    all_paths = sorted(
        path
        for path in tracked | worktree
        if _in_scope(path, scope)
    )
    decisions: dict[str, _Decision] = {
        path: _Decision(
            path,
            "excluded",
            _default_directory_reason(
                path,
                exclude_generated_state=request.scan_policy.exclude_generated_state,
            ),
        )
        for path in sorted(excluded_directories)
    }
    for path in output_exclusions:
        decisions[path] = _Decision(path, "excluded", "workflow_output_path")

    candidates: list[str] = []
    for path in all_paths:
        default_reason = _default_directory_reason(
            path,
            exclude_generated_state=request.scan_policy.exclude_generated_state,
        )
        reason = (
            "workflow_output_path"
            if path in output_exclusions
            else default_reason
        )
        if not reason and request.scan_policy.exclude_credentials:
            reason = _credential_reason(path)
        if not reason and _matches(path, request.scan_policy.exclude_patterns):
            reason = "explicit_exclude_pattern"
        if (
            not reason
            and request.scan_policy.include_patterns
            and not _matches(path, request.scan_policy.include_patterns)
        ):
            reason = "not_selected_by_include_pattern"
        is_untracked = path in worktree and path not in tracked
        if not reason and is_untracked and path in ignored:
            reason = "repository_ignore_policy"
        if (
            not reason
            and is_untracked
            and not request.scan_policy.include_untracked
        ):
            reason = "untracked_not_admitted"
        if not reason and _looks_binary(path):
            reason = "large_or_binary_default"
        if reason:
            decisions[path] = _Decision(
                path,
                "excluded",
                reason,
                ("content_not_read",)
                if reason == "credential_or_key_material"
                else (),
            )
        else:
            candidates.append(path)

    included: list[str] = []
    admitted_digests: set[str] = set()
    admitted_bytes = 0
    inspected_bytes = 0
    git_blob_cache: dict[str, bytes] = {}
    git_blob_sizes: dict[str, int] = {}
    budget_exhausted = False

    def git_blob_size(entry: _GitEntry) -> int:
        cached = git_blob_sizes.get(entry.object_id)
        if cached is not None:
            return cached
        raw = _run_git(
            root,
            ("cat-file", "-s", entry.object_id),
            deadline=deadline,
        )
        try:
            size = int(raw.decode("ascii", "strict").strip())
        except (UnicodeDecodeError, ValueError) as exc:
            raise DirectoryResolutionError(
                "Git returned a malformed blob size"
            ) from exc
        if size < 0:
            raise DirectoryResolutionError("Git returned a negative blob size")
        git_blob_sizes[entry.object_id] = size
        return size

    def git_blob(entry: _GitEntry) -> bytes:
        cached = git_blob_cache.get(entry.object_id)
        if cached is not None:
            return cached
        data = _run_git(
            root,
            ("cat-file", "blob", entry.object_id),
            deadline=deadline,
        )
        git_blob_cache[entry.object_id] = data
        return data

    def matches_git_object(data: bytes, entry: _GitEntry) -> bool:
        framed = f"blob {len(data)}\0".encode("ascii") + data
        digest = (
            # This reproduces Git's object identifier; it is not a security
            # or evidence digest (all scanner evidence uses SHA-256).
            hashlib.sha1(framed).hexdigest()
            if len(entry.object_id) == 40
            else hashlib.sha256(framed).hexdigest()
        )
        return digest == entry.object_id

    for path in candidates:
        _remaining_seconds(deadline)
        if len(included) >= budget.max_files:
            decisions[path] = _Decision(path, "excluded", "max_files_budget")
            continue
        if budget_exhausted:
            decisions[path] = _Decision(path, "excluded", "max_scan_bytes_budget")
            continue
        git_entries = tuple(
            {
                entry.object_id: entry
                for entry in (head.get(path), index.get(path))
                if entry is not None
            }.values()
        )
        if any(git_blob_size(entry) > budget.max_file_bytes for entry in git_entries):
            decisions[path] = _Decision(
                path, "excluded", "max_file_bytes_default"
            )
            continue
        variants: list[bytes] = []
        worktree_size = 0
        if path in worktree:
            absolute = root.joinpath(*PurePosixPath(path).parts)
            try:
                file_stat = absolute.lstat()
            except OSError as exc:
                raise UnstableDirectoryScanError(
                    f"candidate disappeared during scan: {path!r}"
                ) from exc
            if stat.S_ISLNK(file_stat.st_mode):
                if request.scan_policy.reject_symlinks:
                    raise ScanSymlinkError(
                        f"selected symlink is forbidden: {path!r}"
                    )
                resolved = absolute.resolve(strict=False)
                try:
                    if Path(os.path.commonpath((str(root), str(resolved)))) != root:
                        raise ScanSymlinkError(
                            f"selected symlink escapes repository: {path!r}"
                        )
                except ValueError as exc:
                    raise ScanSymlinkError(
                        f"selected symlink escapes repository: {path!r}"
                    ) from exc
            elif not stat.S_ISREG(file_stat.st_mode):
                raise DirectoryResolutionError(
                    f"special file is forbidden: {path!r}"
                )
            if file_stat.st_size > budget.max_file_bytes:
                decisions[path] = _Decision(
                    path, "excluded", "max_file_bytes_default"
                )
                continue
            worktree_size = file_stat.st_size
        if inspected_bytes + worktree_size > budget.max_scan_bytes:
            decisions[path] = _Decision(
                path, "excluded", "max_scan_bytes_inspection_budget"
            )
            budget_exhausted = True
            continue
        if path in worktree:
            absolute = root.joinpath(*PurePosixPath(path).parts)
            worktree_data = _stable_file_bytes(
                absolute, path, budget.max_file_bytes
            )
            variants.append(worktree_data)
            inspected_bytes += len(worktree_data)
            for entry in git_entries:
                if matches_git_object(worktree_data, entry):
                    git_blob_cache.setdefault(entry.object_id, worktree_data)
        git_inspection_charge = sum(
            git_blob_size(entry)
            for entry in git_entries
            if entry.object_id not in git_blob_cache
        )
        if inspected_bytes + git_inspection_charge > budget.max_scan_bytes:
            decisions[path] = _Decision(
                path, "excluded", "max_scan_bytes_inspection_budget"
            )
            budget_exhausted = True
            continue
        for entry in git_entries:
            data = git_blob(entry)
            variants.append(data)
        inspected_bytes += git_inspection_charge
        if not variants:
            continue
        if any(_looks_binary(path, data) for data in variants):
            decisions[path] = _Decision(path, "excluded", "binary_content_default")
            continue
        if any(_contains_secret(data) for data in variants):
            raise SecretLeakageError(
                f"secret-like material detected in admitted path {path!r}"
            )
        new_digests: dict[str, int] = {}
        for data in variants:
            digest = "sha256:" + hashlib.sha256(data).hexdigest()
            if digest not in admitted_digests:
                new_digests[digest] = len(data)
        additional = sum(new_digests.values())
        if admitted_bytes + additional > budget.max_scan_bytes:
            decisions[path] = _Decision(path, "excluded", "max_scan_bytes_budget")
            budget_exhausted = True
            continue
        admitted_digests.update(new_digests)
        admitted_bytes += additional
        included.append(path)
        decisions[path] = _Decision(
            path,
            "included",
            (
                "tracked_or_staged"
                if path in tracked
                else "policy_admitted_untracked"
            ),
            ("source_body_content_addressed",),
        )

    excluded_paths = tuple(
        sorted(
            {
                item.path
                for item in decisions.values()
                if item.disposition == "excluded"
            }
        )
    )
    from ..core.program_behavior import (
        ProgramBehaviorError,
        RepositoryRaceError,
        RequiredInputTooLargeError,
        SnapshotBounds,
        SymlinkEscapeError,
        build_program_behavior,
    )

    observation_bound = min(
        1_000_000,
        max(256, budget.max_symbols * 16, budget.max_files * 4),
    )
    try:
        behavior = build_program_behavior(
            root,
            scopes=(scope,),
            excluded_paths=excluded_paths,
            bounds=SnapshotBounds(
                max_file_bytes=budget.max_file_bytes,
                max_total_bytes=budget.max_scan_bytes,
                max_files=budget.max_files,
                max_observations=observation_bound,
            ),
            artifact_store=artifact_store,
            previous=previous.program_behavior if previous is not None else None,
        )
    except RepositoryRaceError as exc:
        raise UnstableDirectoryScanError(
            "admitted repository bytes changed during scan"
        ) from exc
    except SymlinkEscapeError as exc:
        raise ScanSymlinkError(
            "selected symlink escapes the admitted scan closure"
        ) from exc
    except RequiredInputTooLargeError as exc:
        raise ScanBudgetError(
            "exact program-behavior scan exceeded a declared bound"
        ) from exc
    except ProgramBehaviorError as exc:
        raise PromptDirectoryScanError(
            f"exact program-behavior scan failed ({_safe_reason(type(exc).__name__)})"
        ) from exc

    actual_paths = tuple(item.path for item in behavior.repository.entries)
    if actual_paths != tuple(sorted(included)):
        raise UnstableDirectoryScanError(
            "policy selection and exact worktree snapshot disagree"
        )
    if behavior.repository.stats.hashed_bytes != admitted_bytes:
        raise UnstableDirectoryScanError(
            "preflight bytes and exact worktree snapshot disagree"
        )
    dirty_worktree_root = _cid_for(
        {
            "identity_kind": "program-behavior-dirty-worktree-root",
            "upstream_identity": behavior.dirty_worktree_root,
        }
    )
    program_root = _cid_for(
        {
            "identity_kind": "program-behavior-root",
            "upstream_identity": behavior.behavior_root,
        }
    )
    ast_root = _cid_for(
        {
            "identity_kind": "program-behavior-ast-root",
            "upstream_identity": behavior.analysis.ast_root,
        }
    )

    configuration = {
        "schema": SCAN_CONFIGURATION_SCHEMA,
        "scanner_version": PROMPT_DIRECTORY_SCANNER_VERSION,
        "declared_scanner_version": request.scan_policy.scanner_version,
        "scanner_policy_cid": request.scan_policy.content_id,
        "requested_program_root": request.program_root,
        "repository_root_cid": request.repository_root_cid,
        "allowlist_cid": allowlist.allowlist_cid,
        "scope": scope,
        "output_exclusions": list(output_exclusions),
        "max_evidence_artifact_bytes": MAX_EVIDENCE_ARTIFACT_BYTES,
        "budget": budget.to_dict(),
    }
    scanner_root = _cid_for(
        {
            "schema": SCAN_CONFIGURATION_SCHEMA,
            "implementation_version": PROMPT_DIRECTORY_SCANNER_VERSION,
            "scanner_policy_cid": request.scan_policy.content_id,
        }
    )
    configuration["scanner_root"] = scanner_root
    configuration_root = _cid_for(configuration)
    categories, truncations, category_counts = _classify(
        behavior, included, budget.max_symbols
    )
    artifacts: list[ScanArtifact] = []
    category_artifacts: dict[str, ScanArtifact] = {}
    for name, payload in categories.items():
        artifact = _artifact(
            name,
            {
                "schema": SCAN_EVIDENCE_ARTIFACT_SCHEMA,
                "kind": name,
                "dirty_worktree_root": dirty_worktree_root,
                "configuration_root": configuration_root,
                "summary": payload,
            },
            artifact_store=artifact_store,
        )
        artifacts.append(artifact)
        category_artifacts[name] = artifact

    source_manifest = _artifact(
        "worktree-manifest",
        {
            "schema": SCAN_EVIDENCE_ARTIFACT_SCHEMA,
            "kind": "worktree-manifest",
            "dirty_worktree_root": dirty_worktree_root,
            "entries": [
                {
                    "path": item.path,
                    "status": item.status.value,
                    "kind": item.kind.value,
                    "head_digest": item.head_blob.digest if item.head_blob else "",
                    "index_digest": item.index_blob.digest if item.index_blob else "",
                    "worktree_digest": (
                        item.worktree_blob.digest if item.worktree_blob else ""
                    ),
                    "rename_from": item.rename_from,
                }
                for item in behavior.repository.entries
            ],
        },
        artifact_store=artifact_store,
    )
    artifacts.append(source_manifest)
    decision_manifest = _artifact(
        "scan-decisions",
        {
            "schema": SCAN_DECISION_MANIFEST_SCHEMA,
            "configuration": configuration,
            "configuration_root": configuration_root,
            "decisions": [
                decisions[path].to_dict() for path in sorted(decisions)
            ],
            "stability_checks": [
                "canonical_roots_before_scan",
                "stable_file_reads",
                "program_behavior_post_hash_verification",
                "root_witness_after_analysis",
                "output_paths_after_analysis",
            ],
        },
        artifact_store=artifact_store,
    )
    artifacts.append(decision_manifest)

    context = OptionalAnalysisContext(
        request_cid=request.request_cid,
        repository_root_cid=request.repository_root_cid,
        dirty_worktree_root=dirty_worktree_root,
        scanner_policy_cid=request.scan_policy.content_id,
        program_root=program_root,
        ast_root=ast_root,
        configuration_root=configuration_root,
        included_paths=tuple(sorted(included)),
        category_counts=MappingProxyType(category_counts),
        max_summary_bytes=min(16_384, budget.max_serialized_bytes // 4 or 1),
    )
    optional, optional_status = _optional_result(optional_analysis, context)
    optional_payload = {
        "schema": SCAN_EVIDENCE_ARTIFACT_SCHEMA,
        "kind": "optional-analysis",
        "status": optional.status,
        "summary_digest": _cid_for(
            {"summary": optional.summary, "status": optional.status}
        ),
        "upstream_artifact_cid": optional.artifact_cid,
        "repository_paths": list(optional.repository_paths),
        "claim_keys": list(optional.claim_keys),
        "authority": EvidenceAuthority.SCAN_ADVISORY.value,
        "binding": {
            "repository_root_cid": request.repository_root_cid,
            "dirty_worktree_root": dirty_worktree_root,
            "scanner_policy_cid": request.scan_policy.content_id,
        },
    }
    optional_artifact = _artifact(
        "optional-analysis",
        optional_payload,
        artifact_store=artifact_store,
    )
    artifacts.append(optional_artifact)

    evidence: list[PromptEvidenceRecord] = [
        _evidence_record(
            key=f"scan:{name}",
            kind=f"directory_scan_{name}",
            summary=_category_summary(name, categories[name]),
            artifact=category_artifacts[name],
            paths=(
                [
                    record["path"]
                    for record in categories[name].get("records", ())
                ]
                if name in {"interfaces", "symbols"}
                else [
                    path
                    for value in (
                        categories[name].get("paths", ())
                        if name != "languages"
                        else [
                            path
                            for group in categories[name]["paths"].values()
                            for path in group
                        ]
                    )
                    for path in ([value] if isinstance(value, str) else value)
                ]
            ),
            claims=(f"scan:{name}:advisory",),
        )
        for name in (
            "languages",
            "build",
            "interfaces",
            "symbols",
            "tests",
            "documents",
            "policies",
        )
    ]
    evidence.extend(
        (
            _evidence_record(
                key="scan:worktree-manifest",
                kind="directory_scan_worktree",
                summary=(
                    f"Exact worktree manifest binds {len(actual_paths)} paths and "
                    f"{behavior.repository.stats.hashed_bytes} unique bytes."
                ),
                artifact=source_manifest,
                paths=actual_paths,
                claims=("scan:exact-worktree-root",),
                extra_provenance={
                    "program_root": behavior.behavior_root,
                    "ast_root": behavior.analysis.ast_root,
                },
            ),
            _evidence_record(
                key="scan:decisions",
                kind="directory_scan_policy",
                summary=(
                    f"Policy manifest records {len(decisions)} exact include, "
                    "exclude, and redaction decisions."
                ),
                artifact=decision_manifest,
                claims=("scan:policy-decisions",),
                extra_provenance={
                    "configuration_root": configuration_root,
                    "scanner_root": scanner_root,
                    "scanner_policy_cid": request.scan_policy.content_id,
                },
            ),
            _evidence_record(
                key="scan:optional-analysis",
                kind="directory_scan_optional_analysis",
                summary=optional.summary,
                artifact=optional_artifact,
                paths=optional.repository_paths,
                claims=optional.claim_keys,
                extra_provenance={"degradation_status": optional_status},
            ),
        )
    )
    if len(evidence) > budget.max_evidence:
        aggregate = _artifact(
            "scan-summary",
            {
                "schema": SCAN_EVIDENCE_ARTIFACT_SCHEMA,
                "kind": "scan-summary",
                "evidence_artifact_cids": [
                    item.artifact_cid for item in artifacts
                ],
                "category_counts": category_counts,
                "optional_analysis_status": optional_status,
            },
            artifact_store=artifact_store,
        )
        artifacts.append(aggregate)
        evidence = [
            _evidence_record(
                key="scan:bounded-summary",
                kind="directory_scan_summary",
                summary=(
                    "Bounded aggregate covers language, build, interface, symbol, "
                    "test, document, policy, worktree, and optional-analysis evidence."
                ),
                artifact=aggregate,
                paths=actual_paths,
                claims=tuple(
                    f"scan:{name}:advisory"
                    for name in (
                        "languages",
                        "build",
                        "interfaces",
                        "symbols",
                        "tests",
                        "documents",
                        "policies",
                    )
                ),
                extra_provenance={"collapsed_record_count": 10},
            )
        ]
        truncations.append(
            f"evidence_projection:max_evidence:1-of-10:manifest:{aggregate.artifact_cid}"
        )

    index_root = _cid_for(
        {
            "schema": SCAN_EVIDENCE_ARTIFACT_SCHEMA,
            "kind": "scan-index",
            "configuration_root": configuration_root,
            "ast_root": ast_root,
            "decision_manifest_cid": decision_manifest.artifact_cid,
            "source_manifest_cid": source_manifest.artifact_cid,
            "evidence_cids": sorted(item.evidence_cid for item in evidence),
            "optional_analysis_status": optional_status,
        }
    )
    excluded_decisions = [
        item for item in decisions.values() if item.disposition == "excluded"
    ]
    exclusion_strings: list[str] = []
    exclusion_bytes = 0
    exclusion_limit = max(256, budget.max_serialized_bytes // 4)
    for item in sorted(excluded_decisions, key=lambda value: value.path):
        rendered = f"{item.path}: {item.reason}"
        size = len(rendered.encode("utf-8"))
        if len(exclusion_strings) >= 1_024 or exclusion_bytes + size > exclusion_limit:
            break
        exclusion_strings.append(rendered)
        exclusion_bytes += size
    omitted_exclusions = len(excluded_decisions) - len(exclusion_strings)
    if omitted_exclusions:
        truncations.append(
            "exclusion_projection:"
            f"{omitted_exclusions}-omitted:manifest:{decision_manifest.artifact_cid}"
        )
    if any(
        item.reason
        in {
            "max_files_budget",
            "max_scan_bytes_budget",
            "max_scan_bytes_inspection_budget",
        }
        for item in excluded_decisions
    ):
        truncations.append(
            f"scan_scope:budget_exclusions:manifest:{decision_manifest.artifact_cid}"
        )
    truncations = sorted(set(truncations))
    status_counts = Counter(
        item.status.value for item in behavior.repository.entries
    )
    counts = {
        "files": len(actual_paths),
        "scan_bytes": behavior.repository.stats.hashed_bytes,
        "inspected_bytes": inspected_bytes,
        "symbols": category_counts["symbols"],
        "included": len(included),
        "excluded": len(excluded_decisions),
        "redacted": sum(bool(item.redactions) for item in decisions.values()),
        "tracked": sum(path in tracked for path in included),
        "untracked": status_counts.get("untracked", 0),
        "clean": status_counts.get("clean", 0),
        "modified": status_counts.get("modified", 0),
        "staged": status_counts.get("staged", 0),
        "staged_and_modified": status_counts.get("staged_and_modified", 0),
        "deleted": status_counts.get("deleted", 0)
        + status_counts.get("staged_deletion", 0),
        "renamed": status_counts.get("renamed", 0),
        "evidence": len(evidence),
        **category_counts,
    }
    _verify_root_witness(root, directory, witness, deadline=deadline)
    if _validate_output_paths(request, root, scope) != output_exclusions:
        raise UnstableDirectoryScanError(
            "workflow output path resolution changed during scan"
        )
    try:
        behavior.verify_unchanged()
    except Exception as exc:
        raise UnstableDirectoryScanError(
            "admitted repository bytes changed after analysis"
        ) from exc
    finished_at_ms = int((clock_ms or (lambda: time.time_ns() // 1_000_000))())
    receipt = DirectoryScanReceipt(
        request_cid=request.request_cid,
        repository_root=str(root),
        directory=str(directory),
        repository_root_cid=request.repository_root_cid,
        dirty_worktree_root=dirty_worktree_root,
        scanner_policy_cid=request.scan_policy.content_id,
        program_root=program_root,
        ast_root=ast_root,
        index_root=index_root,
        budget=budget,
        evidence=tuple(evidence),
        counts=counts,
        exclusions=tuple(exclusion_strings),
        truncations=tuple(truncations),
        truncated=bool(truncations),
        started_at_ms=started_at_ms,
        finished_at_ms=finished_at_ms,
    )
    return DirectoryScanDetails(
        receipt=receipt,
        artifacts=tuple(artifacts),
        program_behavior=behavior,
        configuration_root=configuration_root,
        stability_checks=(
            "canonical_roots_before_scan",
            "stable_file_reads",
            "program_behavior_post_hash_verification",
            "root_witness_after_analysis",
            "output_paths_after_analysis",
            "final_rehash",
        ),
        optional_analysis_status=optional_status,
    )


def scan_prompt_directory(
    request: PromptWorkflowRequest,
    *,
    repository_allowlist: RepositoryAllowlist
    | Sequence[str | os.PathLike[str]],
    artifact_store: "BoundedArtifactStore | None" = None,
    optional_analysis: OptionalAnalysisAdapter
    | Callable[[OptionalAnalysisContext], Any]
    | None = None,
    previous: DirectoryScanDetails | None = None,
    clock_ms: Callable[[], int] | None = None,
) -> DirectoryScanReceipt:
    """Return the canonical ASI-142 directory-scan receipt."""

    return scan_prompt_directory_detailed(
        request,
        repository_allowlist=repository_allowlist,
        artifact_store=artifact_store,
        optional_analysis=optional_analysis,
        previous=previous,
        clock_ms=clock_ms,
    ).receipt


class PromptDirectoryScanner:
    """Reusable scanner façade with an explicit, immutable allowlist."""

    def __init__(
        self,
        repository_allowlist: RepositoryAllowlist
        | Sequence[str | os.PathLike[str]],
        *,
        artifact_store: "BoundedArtifactStore | None" = None,
        optional_analysis: OptionalAnalysisAdapter
        | Callable[[OptionalAnalysisContext], Any]
        | None = None,
    ) -> None:
        self.repository_allowlist = _coerce_allowlist(repository_allowlist)
        self.artifact_store = artifact_store
        self.optional_analysis = optional_analysis
        self._previous: DirectoryScanDetails | None = None

    def scan(
        self,
        request: PromptWorkflowRequest,
        *,
        clock_ms: Callable[[], int] | None = None,
    ) -> DirectoryScanReceipt:
        details = self.scan_detailed(request, clock_ms=clock_ms)
        return details.receipt

    def scan_detailed(
        self,
        request: PromptWorkflowRequest,
        *,
        clock_ms: Callable[[], int] | None = None,
    ) -> DirectoryScanDetails:
        details = scan_prompt_directory_detailed(
            request,
            repository_allowlist=self.repository_allowlist,
            artifact_store=self.artifact_store,
            optional_analysis=self.optional_analysis,
            previous=self._previous,
            clock_ms=clock_ms,
        )
        self._previous = details
        return details

    def wire_planning_analysis_factory(self, factory: Any) -> "PromptDirectoryScanner":
        """Attach a production planning factory as the default optional analysis.

        The factory must expose ``optional_analysis`` (PDR-011
        :class:`PlanningAnalysisFactory`).  Admission wiring belongs on the
        prompt supervisor service; this scanner only consumes advisory analysis.
        """

        adapter = getattr(factory, "optional_analysis", None)
        if adapter is None:
            raise TypeError(
                "planning analysis factory must expose optional_analysis"
            )
        self.optional_analysis = adapter
        wire = getattr(factory, "wire_prompt_directory_scanner", None)
        if callable(wire):
            wire(self)
        return self


def build_prompt_scanner_with_planning_factory(
    repository_allowlist: RepositoryAllowlist
    | Sequence[str | os.PathLike[str]],
    factory: Any,
    *,
    artifact_store: "BoundedArtifactStore | None" = None,
) -> PromptDirectoryScanner:
    """Construct a scanner with production planning ``optional_analysis`` wired."""

    scanner = PromptDirectoryScanner(
        repository_allowlist,
        artifact_store=artifact_store,
    )
    return scanner.wire_planning_analysis_factory(factory)


build_directory_scan_receipt = scan_prompt_directory
scan_directory = scan_prompt_directory
DirectoryScanner = PromptDirectoryScanner
DirectoryScanResult = DirectoryScanDetails


__all__ = [
    "ApproximateEvidenceAuthorityError",
    "DirectoryResolutionError",
    "DirectoryScanDetails",
    "DirectoryScanResult",
    "DirectoryScanner",
    "MAX_EVIDENCE_ARTIFACT_BYTES",
    "NestedRepositoryError",
    "OptionalAnalysisAdapter",
    "OptionalAnalysisContext",
    "OptionalAnalysisResult",
    "PROMPT_DIRECTORY_SCANNER_VERSION",
    "PromptDirectoryScanError",
    "PromptDirectoryScanner",
    "RepositoryAllowlist",
    "RepositoryAllowlistError",
    "SCANNER_VERSION",
    "ScanArtifact",
    "ScanBudgetError",
    "ScanSymlinkError",
    "SecretLeakageError",
    "UnstableDirectoryScanError",
    "build_directory_scan_receipt",
    "build_prompt_scanner_with_planning_factory",
    "build_repository_allowlist",
    "repository_allowlist_cid",
    "repository_root_cid",
    "scan_directory",
    "scan_prompt_directory",
    "scan_prompt_directory_detailed",
]
